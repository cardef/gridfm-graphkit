#!/bin/bash
#SBATCH --job-name=gridfm-fm-scaling
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=mlruns/fm-scaling/slurm-logs/%x-%j.out

set -euo pipefail

: "${CAMPAIGN_MANIFEST:?set CAMPAIGN_MANIFEST to the frozen manifest}"
: "${RUN_ID:?set RUN_ID to one explicit E001-E020 identifier}"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

../.venv/bin/python -m experiments.fm_scaling.launch \
  --manifest "$CAMPAIGN_MANIFEST" \
  --run-id "$RUN_ID" \
  --repo-root "$repo_root" \
  --python ../.venv/bin/python \
  --execute
