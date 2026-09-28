# Results

`leaderboard.csv` lists every public-leaderboard upload in order, with the holdout score where one was measured.

- **holdout_f05**: macro F0.5 on the 10% holdout of training entities (for the final two rows: the unseen half of the
  holdout, after the decoder was tuned on the other half). Rows with a look-alike shift report the holdout *with* that
  shift applied, which is lower than without it; the leaderboard rewarded the shift because the test set contains far
  more neighbour look-alike businesses than the training data.
- **kind = probe**: files built on purpose to measure one thing (e.g. a country's contribution), not to score well.

Main lessons from the table:
1. Holdout and leaderboard disagree once the train/test shift matters (tier5: +0.0026 holdout, −0.0007 leaderboard).
2. The look-alike correction has a clear optimum around 2.5 log-odds (1.2 → 0.9645, 2.5 → 0.9666, 3.5 → 0.9651).
3. Calibrated expected-F0.5 decoding with an explicit "no match" option gave the final gain (0.9666 → 0.9672).
