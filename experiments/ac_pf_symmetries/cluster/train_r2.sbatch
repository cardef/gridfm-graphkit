#!/bin/bash
#SBATCH --job-name=symfm-r2
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --constraint=sapphire
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%A_%a.out
# R2 (multigrid.py): 2 folds x 2 arms x 3 seeds, task i -> fold i/6, then (arm, seed) with the arms alternating
# so every wave holds both. CPU only, one node type (abacus-004, Sapphire Rapids): GPUs were saturated, and mixing
# hardware between arms would confound them.
#   sbatch --array=0-11%4 train_r2.sbatch
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=${CHECKOUT:-$S/SymmetricalFM/gridfm-graphkit}
source $S/env/activate.sh $R
cd $R
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
TRAIN=("case14_ieee case57_ieee case118_ieee" "case14_ieee case30_ieee case118_ieee")
HELDOUT=(case30_ieee case57_ieee)
ARMS=(canon m1fullcanon)
i=$SLURM_ARRAY_TASK_ID
fold=$((i / 6))
j=$((i % 6))
arm=${ARMS[$((j % 2))]}
seed=$((j / 2))
echo "HEAD $(git rev-parse HEAD) checkout=$R fold=$fold arm=$arm seed=$seed host=$(hostname)"
bash experiments/ac_pf_symmetries/cluster/provenance.sh \
    experiments/ac_pf_symmetries/{multigrid,hodge,run,symmetries}.py
python -m experiments.ac_pf_symmetries.multigrid --arm "$arm" --seeds "$seed" --train ${TRAIN[$fold]} \
    --heldout "${HELDOUT[$fold]}" --device cpu \
    --results-dir $S/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/results/abacus_r2
echo R2_DONE
