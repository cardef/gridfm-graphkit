#!/bin/bash
#SBATCH --job-name=symfm-keval
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%A_%a.out
# Piano K K3: the trained models zero-shot on the other N-k sets (k = 2, 3, 5, 10, 20), one result_extrap file per
# model; then PH (fit_vref.py on case118_k's train split, r2_classical.py --vref-coef) on every N-k set.
# Task i -> arm i/3, seed i%3 (as train_k.sbatch); task 0 also fits and runs PH. Needs train_k.sbatch to have finished.
#   sbatch --dependency=afterok:<train array id> --array=0-11%6 eval_k.sbatch
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
ARMS=(plain dccanon dcinjcanon gp)
i=$SLURM_ARRAY_TASK_ID
arm=${ARMS[$((i / 3))]}
seed=$((i % 3))
OUT=$R/experiments/ac_pf_symmetries/results/pianok
echo "HEAD $(git rev-parse HEAD) arm=$arm seed=$seed host=$(hostname) $(date -Is)"
bash experiments/ac_pf_symmetries/cluster/provenance.sh \
    experiments/ac_pf_symmetries/{multigrid,hodge,run,symmetries,dcprior,resnorm,r2_classical,fit_vref}.py
python -m experiments.ac_pf_symmetries.multigrid --eval-only --extrap case118_nk2 case118_nk3 case118_nk5 case118_nk10 case118_nk20 \
    --arm "$arm" --seeds "$seed" --train case118_k --heldout case118_nk1 \
    --split case118_k=data/case118_k/splits/official_seed0.json --device cpu --results-dir $OUT
if [ "$i" = 0 ]; then
    python -m experiments.ac_pf_symmetries.fit_vref case118_k --split data/case118_k/splits/official_seed0.json --out $OUT/vref_coef.json
    python -m experiments.ac_pf_symmetries.r2_classical --vref-coef $OUT/vref_coef.json \
        case118_nk1 case118_nk2 case118_nk3 case118_nk5 case118_nk10 case118_nk20 > $OUT/CLASSICAL_nk_ph.txt
fi
echo K_EVAL_DONE
