#!/bin/bash
# Merge the 4 case2000 shards into data/case2000_goc/raw (see
# experiments/m1/merge_case2000_shards.py for provenance + invariants).
#
#SBATCH --job-name=m1-merge-c2k
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=mlruns/slurm-logs/%x_%j.out

set -euo pipefail
HERE="$SLURM_SUBMIT_DIR/experiments/slurm"
source "$HERE/cluster.env"
cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

python experiments/m1/merge_case2000_shards.py
