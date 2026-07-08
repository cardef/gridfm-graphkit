#!/bin/bash
# R006 (M0 gate, supersedes plain R002): rebuild every Kron-Schur hierarchy
# cache through the amended P-mass + reconstruction gate on the freshly
# generated M1 data, then re-run the R007 case2000 outlier diagnostic, then
# emit the R010 matched-FLOP training matrix (which reads the just-rebuilt
# data/<grid>/processed/ caches for accurate nnz_prolong — this is why
# config-gen happens HERE and not earlier).
#
# KNOWN GAP (not scripted, because no code implements it yet): R007's
# diagnosis is "flag scenarios where a zero-load bus has |V|<0.1 p.u." but
# nothing in gridfm_graphkit or these experiment scripts actually drops
# those scenarios — r007_outlier_triage.py only diagnoses and prints a
# verdict. Read experiments/m0/results/r007_outlier_triage.json after this
# job and decide/implement the filter before trusting case2000 results.
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
#SBATCH --output=experiments/slurm/logs/%x_%j.out

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.env"
cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

python experiments/m0/r006_p_recon_gate.py
python experiments/m0/r007_outlier_triage.py

python experiments/m1/r010_make_configs.py --seeds 0 1 2

N=$(ls experiments/m1/configs/*.yaml | wc -l | tr -d ' ')
echo "R010 config count: $N -> submitting training array 1-$N"
sbatch --account="$SLURM_ACCOUNT" --partition="$PART_GPU" --gres="$GRES_GPU" \
  --array="1-${N}%6" "$HERE/03_train_r010.sbatch"
