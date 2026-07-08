#!/bin/bash
# R011 phase B: GRIT full-attention peak-memory sweep (B5 memory wall),
# batch sizes {1,2,4,8}, stop at first OOM. CPU-only phase A runs inline
# too (script does both per grid). Only needs data/<grid>/raw from
# 01_datagen — no Kron-Schur hierarchy, so no dependency on 02.
#
#SBATCH --job-name=m1-r011-phaseb
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/slurm/logs/%x_%j.out

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.env"
cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

python experiments/m1/r011_grit_memory.py --batch-sizes 1 2 4 8
