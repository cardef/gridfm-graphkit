#!/bin/bash
# R010 training matrix: one array task per experiments/m1/configs/*.yaml
# (ks_<grid>, flat_<grid>_d{8,16,32,48}[_iso], seeds 0/1/2 x2 grids —
# case500_goc, case2000_goc). Submitted by 02_precompute_gate.sbatch with
# the array range sized to the actual config count; --array here is a
# fallback for standalone submission only (recount after 02 has run).
#
# --compile mode: experiments/m1/README.md's engineering note (E4) says
# "max-autotune at case2000, reduce-overhead at <=case118" without stating
# case500 explicitly. INFERRED (not verified against a source): case500
# takes reduce-overhead, case2000 takes max-autotune. Override below if
# that reading is wrong.
#
#SBATCH --job-name=m1-r010
#SBATCH --array=1-18%6
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=experiments/slurm/logs/%x_%A_%a.out

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.env"
cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

CFG=$(ls experiments/m1/configs/*.yaml | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$CFG" ]; then
  echo "no config at index $SLURM_ARRAY_TASK_ID (only $(ls experiments/m1/configs/*.yaml | wc -l) exist)"
  exit 1
fi
RUN_NAME=$(basename "$CFG" .yaml)

COMPILE_MODE=reduce-overhead
case "$RUN_NAME" in
  *case2000*) COMPILE_MODE=max-autotune ;;
esac

echo "=== $RUN_NAME ($CFG) compile=$COMPILE_MODE start $(date) ==="
gridfm_graphkit train \
  --config "$CFG" \
  --data_path data \
  --exp_name m1_r010 \
  --run_name "$RUN_NAME" \
  --compile "$COMPILE_MODE"
echo "=== $RUN_NAME done $(date) ==="
