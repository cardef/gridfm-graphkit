#!/bin/bash
# R006 (M0 gate, supersedes plain R002): rebuild every Kron-Schur hierarchy
# cache through the amended P-mass + reconstruction gate on the freshly
# generated M1 data, then re-run the R007 case2000 outlier diagnostic, then
# emit the R010 matched-FLOP training matrix (which reads the just-rebuilt
# data/<grid>/processed/ caches for accurate nnz_prolong — this is why
# config-gen happens HERE and not earlier).
#
# R007 filter status: implemented in gridfm_graphkit/datasets/powergrid_hetero_dataset.py
# (DEGENERATE_VM_THRESHOLD, _degenerate_scenarios, wired into both
# HeteroGridDatasetDisk/MMap.__init__ via PyG's _indices subsetting) —
# commit 42fa4ed, predates this script. Verified 2026-07-08 (4/4 tests
# green, tests/test_degenerate_scenario_filter.py). Fires automatically
# whenever data/<grid>/raw is loaded, no separate step needed here.
#
# Ends by submitting 03_train_r010.sbatch itself, once the config count is
# known (SLURM array sizes are static at submission time, but the R010
# config count depends on data this job just produced — chaining an sbatch
# call from inside a running job is the standard SLURM idiom for this).
#
#SBATCH --job-name=m1-precompute-gate
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=mlruns/slurm-logs/%x_%j.out

set -euo pipefail
# NOTE: sbatch copies this script into /var/spool/slurmd/jobNNN/ before
# running it, so BASH_SOURCE-based self-location resolves to the spool dir,
# not the repo. Use SLURM_SUBMIT_DIR (the cwd sbatch was invoked from)
# instead -- these scripts are always submitted from $REPO_ROOT.
HERE="$SLURM_SUBMIT_DIR/experiments/slurm"
source "$HERE/cluster.env"
cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

python experiments/m0/r006_p_recon_gate.py
python experiments/m0/r007_outlier_triage.py

python experiments/m1/r010_make_configs.py --seeds 0 1 2

N=$(ls experiments/m1/configs/*.yaml | wc -l | tr -d ' ')
echo "R010 config count: $N -> submitting training array 1-$N"
sbatch --account="$SLURM_ACCOUNT" --qos="$SLURM_QOS" --partition="$PART_GPU" --gres="$GRES_GPU" \
  --array="1-${N}%6" "$HERE/03_train_r010.sbatch"
