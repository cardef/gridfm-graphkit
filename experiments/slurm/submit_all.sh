#!/bin/bash
# Submit the M1 cluster pipeline: 01 datagen -> 02 precompute/gate
# (which itself submits 03 training once config count is known) -> 04
# R011 phase B (parallel to 02/03, only needs 01).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.env"

COMMON=(--account="$SLURM_ACCOUNT")

J1=$(sbatch --parsable "${COMMON[@]}" --partition="$PART_CPU" "$HERE/01_datagen.sbatch")
echo "01_datagen: $J1 (array 0-6)"

J2=$(sbatch --parsable "${COMMON[@]}" --partition="$PART_CPU" \
  --dependency=afterok:$J1 "$HERE/02_precompute_gate.sbatch")
echo "02_precompute_gate: $J2 (submits 03_train_r010 on completion)"

J4=$(sbatch --parsable "${COMMON[@]}" --partition="$PART_GPU" --gres="$GRES_GPU" \
  --dependency=afterok:$J1 "$HERE/04_train_r011_phaseb.sbatch")
echo "04_train_r011_phaseb: $J4"
