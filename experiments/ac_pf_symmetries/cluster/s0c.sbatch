#!/bin/bash
#SBATCH --job-name=symfm-s0c
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=60
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%j.out
# Piano S, S0c (CPU): the S0 estimator (step halving) from the labelled state plus perturbations, and from two mixed
# starts, ten configurations in parallel (s0_wls.py).   sbatch --nodelist=abacus-004 s0c.sbatch
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
echo "HEAD $(git rev-parse HEAD) host=$(hostname) $(date -Is)"
O=experiments/ac_pf_symmetries/results/pianos/s0c
mkdir -p $O
G="case118_tr case500_goc case2000_goc"
common="--n-eval 64 --coverage 0.4 0.2 --max-it 50"
run() { python -m experiments.ac_pf_symmetries.s0_wls $G --out $O/$1 $common "${@:2}" > $O.$1.log 2>&1; }
for kind in iid offset; do
    for lv in "0.3 0.003" "1 0.01" "3 0.03" "10 0.1"; do
        set -- $lv
        run ${kind}_$1 --start perturb --perturb-kind $kind --perturb-deg $1 --perturb-vm $2 &
    done
done
run dc_vtrue --start dc_vtrue &
run true_vflat --start true_vflat &
wait
echo S0C_DONE
