#!/bin/bash
#SBATCH --job-name=symfm-c14
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%A_%a.out
# Cluster replicate of the case14 campaign: one (variant, seed) per array task, task i -> VARIANTS[i % n],
# SEEDS[i / n]. Same protocol as the PoC (run.py defaults); results under results/abacus/.
#   sbatch --array=0-26%5 train_case14.sbatch
# T4 on purpose: zero billing weight on this cluster, so the campaign costs no fair-share.
# CHECKOUT=<other checkout> runs that code (e.g. the dev copy while queued jobs still import this one's);
# results always land in this checkout's results/abacus.
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=${CHECKOUT:-$S/SymmetricalFM/gridfm-graphkit}
source $S/env/activate.sh $R
cd $R
VARIANTS=(${VARIANTS:-m0 canon aug augcov augnophys m0nophys augmild augphase augscale})
SEEDS=(${SEEDS:-0 1 2})
i=$SLURM_ARRAY_TASK_ID
v=${VARIANTS[$((i % ${#VARIANTS[@]}))]}
s=${SEEDS[$((i / ${#VARIANTS[@]}))]}
OUT=$S/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/results/abacus
echo "HEAD $(git rev-parse HEAD) checkout=$R variant=$v seed=$s host=$(hostname)"
bash experiments/ac_pf_symmetries/cluster/provenance.sh experiments/ac_pf_symmetries/{run,symmetries}.py
nvidia-smi -L
python -m experiments.ac_pf_symmetries.run --variant "$v" --seeds "$s" --results-dir "$OUT" ${EXTRA_ARGS:-}
echo TRAIN_DONE
