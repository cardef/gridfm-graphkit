#!/usr/bin/env bash
#SBATCH --job-name=fm-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=mlruns/fm-scaling/slurm-logs/%x-%A_%a.out

set -euo pipefail

: "${CAMPAIGN_MANIFEST:?set CAMPAIGN_MANIFEST}"
: "${RUN_ID:?set RUN_ID}"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
../.venv/bin/python -m experiments.fm_scaling.evaluate_campaign \
  --manifest "${CAMPAIGN_MANIFEST}" \
  --run-id "${RUN_ID}" \
  --execute
