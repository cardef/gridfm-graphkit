#!/bin/bash
#SBATCH --job-name=symfm-s0b
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%j.out
# Piano S, S0b (CPU): the AC WLS of S0 from the flat, the DC and the labelled (oracle) start, with step halving and
# with Levenberg-Marquardt, six configurations in parallel (s0_wls.py).
#   sbatch --nodelist=abacus-004 s0b.sbatch
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "HEAD $(git rev-parse HEAD) host=$(hostname) $(date -Is)"
O=experiments/ac_pf_symmetries/results/pianos/s0b
for start in flat dc truth; do
    for rule in halving lm; do
        extra=""; [ $rule = lm ] && extra="--lm"
        python -m experiments.ac_pf_symmetries.s0_wls case118_tr case500_goc case2000_goc --out $O/${start}_${rule} \
            --n-eval 64 --coverage 0.6 0.4 0.2 --start $start --max-it 50 $extra > $O.${start}_${rule}.log 2>&1 &
    done
done
wait
echo S0B_DONE
