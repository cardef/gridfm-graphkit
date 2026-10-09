#!/bin/bash
#SBATCH --job-name=symfm-xeval
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%A_%a.out
# Size extrapolation (P47-P50): the saved models of every arm and fold, evaluated zero-shot and unchanged on the grids
# in GRIDS, 1024 fixed scenarios each (multigrid.py --eval-only --extrap, writes result_extrap_<grids>.json in the
# registered run dir). Task i -> fold i/10, arm i%10; the three seeds run in sequence; arms without a saved model in a
# fold (canonresnorm outside fold 3) are skipped. Evaluation only, so the node does not matter. Needs
# build_extrap_data.sbatch to have finished (XDATA_DONE in its log; checked through the processed-cache marker).
#   GRIDS=case500_goc sbatch --array=0-29%15 eval_extrap.sbatch
#   GRIDS=case2000_goc sbatch --array=0-29%10 eval_extrap.sbatch
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
TRAIN=("case14_ieee case57_ieee case118_ieee" "case14_ieee case30_ieee case118_ieee" "case14_ieee case30_ieee case57_ieee")
HELDOUT=(case30_ieee case57_ieee case118_ieee)
ARMS=(canon m1fullcanon m3canon canonresnorm dccanon dczcanon dcinjcanon canonvref dcinjvref dcinj1hop)
OUT=$R/experiments/ac_pf_symmetries/results/abacus_r2
i=$SLURM_ARRAY_TASK_ID
fold=$((i / 10))
arm=${ARMS[$((i % 10))]}
for g in $GRIDS; do
    [ -f "data/$g/processed/processed_raw_files.done" ] || { echo "data of $g not built"; exit 1; }
done
[ -f "$OUT/heldout_${HELDOUT[$fold]}/${arm}_seed0/model/best_model_state_dict.pt" ] || { echo "no $arm model in fold $fold: skip"; exit 0; }
echo "HEAD $(git rev-parse HEAD) fold=$fold arm=$arm grids=$GRIDS host=$(hostname)"
bash experiments/ac_pf_symmetries/cluster/provenance.sh \
    experiments/ac_pf_symmetries/{multigrid,hodge,run,symmetries,dcprior,resnorm}.py
python -m experiments.ac_pf_symmetries.multigrid --eval-only --extrap $GRIDS --arm "$arm" --seeds 0 1 2 \
    --train ${TRAIN[$fold]} --heldout "${HELDOUT[$fold]}" --device cpu --results-dir $OUT
echo XEVAL_DONE
