#!/bin/bash
#SBATCH --job-name=symfm-k
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%A_%a.out
# Piano K training (K2): arms plain (G0), dccanon (+dc), dcinjcanon (+dcinj), gp (+vref +canon +resnorm +dc) on the
# official seed-0 split of the public N-1 case118 set (data/case118_k), zero-shot on case118_nk1 at the end; the other
# N-k sets are evaluated afterwards with --eval-only --extrap. Task i -> arm i/3, seed i%3. Needs build_k_data.sbatch.
#   sbatch --array=0-11%12 train_k.sbatch
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
[ -f data/case118_k/processed/processed_raw_files.done ] || { echo "data/case118_k not built"; exit 1; }
echo "HEAD $(git rev-parse HEAD) arm=$arm seed=$seed host=$(hostname) $(date -Is)"
bash experiments/ac_pf_symmetries/cluster/provenance.sh \
    experiments/ac_pf_symmetries/{multigrid,hodge,run,symmetries,dcprior,resnorm}.py
python -m experiments.ac_pf_symmetries.multigrid --arm "$arm" --seeds "$seed" --train case118_k --heldout case118_nk1 \
    --split case118_k=data/case118_k/splits/official_seed0.json --device cpu \
    --results-dir $R/experiments/ac_pf_symmetries/results/pianok
echo K_TRAIN_DONE
