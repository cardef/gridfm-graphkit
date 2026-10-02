#!/bin/bash
#SBATCH --job-name=symfm-split
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%A_%a.out
# E0p-dup per grid (array task = grid): duplicates in the full public data, train/test twins per released split.
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
CASES=(case14_ieee case30_ieee case57_ieee case118_ieee)
c=${CASES[$SLURM_ARRAY_TASK_ID]}
bash experiments/ac_pf_symmetries/cluster/provenance.sh experiments/ac_pf_symmetries/split_audit.py
python -m experiments.ac_pf_symmetries.split_audit --case "$c" --out "experiments/ac_pf_symmetries/results/genco/split_audit_$c.json"
echo SPLIT_DONE
