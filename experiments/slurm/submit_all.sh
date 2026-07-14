#!/bin/bash
# Submit the M1 cluster pipeline:
#   01  datagen (7 grids, array 0-6)          ---\
#   01b case2000_shards (4 shards) -> 05 merge ---+--> 02 precompute/gate
#                                                  |    (submits 03 training once config count is known)
#                                                  \--> 04 R011 phase B
# case2000_goc is generated via 01b (sharded across 4 nodes) instead of as
# part of 01's array -- a single-node run was measured at ~7 days (see
# 01b_case2000_shards.sbatch); 02/04 need the merged result, so both wait on
# 05 in addition to 01.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.env"

COMMON=(--account="$SLURM_ACCOUNT" --qos="$SLURM_QOS")

J1=$(sbatch --parsable "${COMMON[@]}" --partition="$PART_CPU" "$HERE/01_datagen.sbatch")
echo "01_datagen: $J1 (array 0-6)"

J1B=$(sbatch --parsable "${COMMON[@]}" --partition="$PART_CPU" "$HERE/01b_case2000_shards.sbatch")
echo "01b_case2000_shards: $J1B (array 0-3)"

J5=$(sbatch --parsable "${COMMON[@]}" --partition="$PART_CPU" \
  --dependency=afterok:$J1B "$HERE/05_merge_c2k.sbatch")
echo "05_merge_c2k: $J5"

J2=$(sbatch --parsable "${COMMON[@]}" --partition="$PART_CPU" \
  --dependency=afterok:$J1,afterok:$J5 "$HERE/02_precompute_gate.sbatch")
echo "02_precompute_gate: $J2 (submits 03_train_r010 on completion)"

J4=$(sbatch --parsable "${COMMON[@]}" --partition="$PART_GPU" --gres="$GRES_GPU" \
  --dependency=afterok:$J1,afterok:$J5 "$HERE/04_train_r011_phaseb.sbatch")
echo "04_train_r011_phaseb: $J4"
