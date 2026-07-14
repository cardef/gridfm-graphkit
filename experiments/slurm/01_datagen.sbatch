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
# NOTE: sbatch copies this script into /var/spool/slurmd/jobNNN/ before
# running it, so BASH_SOURCE-based self-location resolves to the spool dir,
# not the repo. Use SLURM_SUBMIT_DIR (the cwd sbatch was invoked from)
# instead -- these scripts are always submitted from $REPO_ROOT.
HERE="$SLURM_SUBMIT_DIR/experiments/slurm"
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

# This node's system libstdc++ (Rocky Linux 8.10, GCC 8.5.0) only provides up
# to GLIBCXX_3.4.25. Loading Julia (via PythonCall/juliacall) *embedded inside
# this Python process* pulls in the system libstdc++ first (already resident
# from Python/numpy's own C extensions), shadowing Julia's own bundled newer
# copy -- libjulia-internal.so then fails with "GLIBCXX_3.4.26 not found".
# Preloading Julia's bundled libstdc++ before Python starts fixes it (a
# standalone `julia` binary doesn't hit this -- only the embedded-in-Python
# path does). Verified 2026-07-08 on abacus-004.
JULIA_STDCXX=$(ls -d "$HOME"/.julia/juliaup/julia-*/lib/julia/libstdc++.so.6 2>/dev/null | sort -V | tail -1)
if [ -n "$JULIA_STDCXX" ]; then
  export LD_PRELOAD="${JULIA_STDCXX}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

echo "=== $GRID start $(date) (task $SLURM_ARRAY_TASK_ID) ==="
gridfm_datakit generate "experiments/m1/datakit_configs/$GRID.yaml"
echo "=== $GRID done $(date) ($(du -sh "data/$GRID" 2>/dev/null | cut -f1)) ==="
