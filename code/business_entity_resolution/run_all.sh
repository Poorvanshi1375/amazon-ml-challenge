#!/bin/bash
# End-to-end pipeline: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
# usage: ER_DATA_DIR=/path/to/dataset bash run_all.sh
#   ER_DATA_DIR  folder containing train/ and test/ (default: ../../student_resource/dataset)
#   ER_ART_DIR   intermediate files (default: ./artifacts)   ER_OUT_DIR  outputs (default: ./output)
#   ER_N_JOBS    CPU processes (default: all cores)
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PYTHON:-python3}
step() { echo "[$(date +%H:%M:%S)] $*"; }

step "1/11 prepare: normalisation, learned word dictionary, learned address rewrites, roles";  $PY prepare.py
step "2/11 blocking (train): TF-IDF search, 3 indexes, learned state partitions";              $PY blocking.py --split train
step "3/11 blocking (test)";                                                                     $PY blocking.py --split test
step "4/11 pruning: learned meta-blocking model + recall-safe cut-off";                          $PY prune.py --tune --learned
step "5/11 pruning (test) -> final candidate set";                                               $PY prune.py --split test
step "6/11 pair features (train)";                                                               $PY features.py --split train
step "7/11 matcher: LightGBM on complete-group S1 entities + threshold tuning on holdout";       $PY train.py
step "8/11 pair features (test)";                                                                $PY features.py --split test
step "9/11 predict: stage-1 matcher on test";                                                    $PY predict.py
step "10/11 stage-2 sibling model (collective features from stage-1 probabilities)";            $PY stage2.py
step "11/11 blend + look-alike shift + calibration + expected-F0.5 decoding -> both TSVs";       $PY finalize.py
step "done: $(ls -1 "${ER_OUT_DIR:-../output}")"
