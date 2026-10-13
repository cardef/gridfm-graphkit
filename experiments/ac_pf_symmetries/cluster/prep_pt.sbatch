#!/bin/bash
#SBATCH --job-name=symfm-ptprep
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%j.out
# Piano X, X3 (CPU): renumber the 31 public subgrids and the case118 transfer-evaluation subset (prep_pt.py), build the
# case118_tr cache on GPFS, the physics references on it (frontier.py), and GENCO's public transfer checkpoints on the
# fixed zero-shot sets (eval_public.py). Needs fetch_pretrain.sh to have finished. Computed before any X3 model exists.
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
grep -q FETCH_DONE experiments/ac_pf_symmetries/cluster/logs/fetch_pretrain.log || { echo "fetch not done"; exit 1; }
echo "HEAD $(git rev-parse HEAD) host=$(hostname) $(date -Is)"
bash experiments/ac_pf_symmetries/cluster/provenance.sh \
    experiments/ac_pf_symmetries/{prep_pt,eval_public,frontier,multigrid,run,subset_data}.py
python -m experiments.ac_pf_symmetries.prep_pt
OUT=experiments/ac_pf_symmetries/results/pianox_pt
mkdir -p $OUT
python -m experiments.ac_pf_symmetries.frontier case118_tr --methods fdpf --no-backbone --tag pt --out $OUT \
    > $OUT/FRONTIER_case118_tr.txt
python -m experiments.ac_pf_symmetries.eval_public --device cpu > $OUT/PUBLIC.txt
echo PTPREP_DONE
