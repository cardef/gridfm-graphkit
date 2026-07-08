#!/bin/bash
# R001 (M1, cluster scale): datakit generation, one array task per grid.
# Mirrors experiments/m0/r001_run_all.sh (local, sequential) but parallel
# across a SLURM array; configs are experiments/m1/datakit_configs/*.yaml
# (experiments/m1/r001_cluster_configs.py — ~10k solved scenarios/grid,
# case2000 oversubscribed 4x for its measured 25% fast-PF yield).
#
#SBATCH --job-name=m1-datagen
#SBATCH --array=0-6
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/slurm/logs/%x_%A_%a.out

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.env"
cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

# Order matches experiments/m1/r001_cluster_configs.py GRIDS.
GRIDS=(case14_ieee case30_ieee case57_ieee case118_ieee case500_goc case2000_goc Texas2k_case1_2016summerpeak)
GRID="${GRIDS[$SLURM_ARRAY_TASK_ID]}"

# CAUTION: first-ever run on a fresh JULIA_DEPOT_PATH triggers Julia package
# precompilation. 7 array tasks hitting an empty shared depot at once can
# race/corrupt it. Warm the depot once first, e.g.:
#   sbatch --array=0-0 experiments/slurm/01_datagen.sbatch
# then submit the full 0-6 array once that task's log shows precompilation
# finished.

echo "=== $GRID start $(date) (task $SLURM_ARRAY_TASK_ID) ==="
gridfm_datakit generate "experiments/m1/datakit_configs/$GRID.yaml"
echo "=== $GRID done $(date) ($(du -sh "data/$GRID" 2>/dev/null | cut -f1)) ==="
