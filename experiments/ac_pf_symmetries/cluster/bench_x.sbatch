#!/bin/bash
#SBATCH --job-name=symfm-bench
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%j.out
# Piano X, X2: inference timing of plain / dcinjcanon / dcloopgcanon on one GPU (bench_x.py).
#   sbatch --nodelist=abacus-006 --gres=gpu:h100:1 --export=ALL,TAG=h100 bench_x.sbatch
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
echo "HEAD $(git rev-parse HEAD) host=$(hostname) $(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
python -m experiments.ac_pf_symmetries.bench_x --out experiments/ac_pf_symmetries/results/pianox/bench_${TAG}.json 2>&1 | grep -v Warning
echo BENCH_DONE
