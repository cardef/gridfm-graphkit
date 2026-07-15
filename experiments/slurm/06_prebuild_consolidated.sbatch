#!/bin/bash
# Pre-build consolidated.pt for case500_goc + case2000_goc, once, before
# the R010 array launches. See experiments/m1/prebuild_consolidated.py
# for why: 6 concurrent training tasks each racing to build this missing
# cache all OOM'd at --mem=64G.
#
#SBATCH --job-name=m1-prebuild-consolidated
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=mlruns/slurm-logs/%x_%j.out

set -euo pipefail
HERE="$SLURM_SUBMIT_DIR/experiments/slurm"
source "$HERE/cluster.env"
cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

python experiments/m1/prebuild_consolidated.py case500_goc case2000_goc
