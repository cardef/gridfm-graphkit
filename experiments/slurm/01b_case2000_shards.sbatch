#!/bin/bash
# case2000_goc cluster datagen, sharded: 4 array tasks (seeds 0-3, 10240
# requested scenarios each = the original 40960). Single-node at 16 procs
# measured ~14 s/scenario (fast-PF fails on most stressed scalings -> Ipopt
# fallback; M0 measured 25% fast-PF yield) -> ~7 days, so shard x4 nodes
# and use 48 procs (near-linear: independent PF solves). Merge with
# experiments/m1/merge_case2000_shards.py (05_merge sbatch) afterward.
#
#SBATCH --job-name=m1-datagen-c2k
#SBATCH --array=0-3
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/slurm/logs/%x_%A_%a.out

set -euo pipefail
HERE="$SLURM_SUBMIT_DIR/experiments/slurm"
source "$HERE/cluster.env"
cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

JULIA_STDCXX=$(ls -d "$HOME"/.julia/juliaup/julia-*/lib/julia/libstdc++.so.6 2>/dev/null | sort -V | tail -1)
if [ -n "$JULIA_STDCXX" ]; then
  export LD_PRELOAD="${JULIA_STDCXX}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

CFG="experiments/m1/datakit_configs/case2000_shards/shard${SLURM_ARRAY_TASK_ID}.yaml"
echo "=== case2000 shard $SLURM_ARRAY_TASK_ID start $(date) ==="
gridfm_datakit generate "$CFG"
echo "=== case2000 shard $SLURM_ARRAY_TASK_ID done $(date) ==="
