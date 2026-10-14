#!/bin/bash
#SBATCH --job-name=symfm-s0
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%j.out
# Piano S, S0 (CPU): physics-only state estimation on the unseen grids, coverage 1.0-0.2 (s0_wls.py).
#   sbatch --nodelist=abacus-004 s0.sbatch
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "HEAD $(git rev-parse HEAD) host=$(hostname) $(date -Is)"
python -m experiments.ac_pf_symmetries.s0_wls case14_ieee case30_ieee case57_ieee case118_ieee case118_tr case500_goc \
    case2000_goc --out experiments/ac_pf_symmetries/results/pianos/s0 2>&1 | grep -v Warning
echo S0_DONE
