#!/bin/bash
#SBATCH --job-name=symfm-kaudit
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%A_%a.out
# Piano K K1: the released GENCO case118 checkpoints (base/small/tiny, seed 0; trained on the public N-1 set) zero-shot
# on our N-k sets, with their own normalizer stats, all scenarios but 20 as the test set (splits/audit_all.json);
# the datamodule reads <data-path>/<network>/raw, so --data-path is the parent dir (job 63094 failed on that).
# Task i -> size i/6, N-k set i%6. Needs build_k_data.sbatch.
#   sbatch --array=0-17%9 audit_k.sbatch
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
SIZES=(base small tiny)
NK=(case118_nk1 case118_nk2 case118_nk3 case118_nk5 case118_nk10 case118_nk20)
i=$SLURM_ARRAY_TASK_ID
size=${SIZES[$((i / 6))]}
g=${NK[$((i % 6))]}
ck=data/genco_ckpt/case118_ieee/$size/seed0
cfg=$ck/case118_ieee_${size}_seed0.yaml
OUT=$R/experiments/ac_pf_symmetries/results/pianok/genco_released
mkdir -p $OUT
echo "HEAD $(git rev-parse HEAD) size=$size set=$g cfg=$cfg host=$(hostname)"
python -m experiments.ac_pf_symmetries.genco_audit --config "$cfg" --data-path data --model-path $ck/best_model_state_dict.pt \
    --normalizer-stats $ck/normalizer_stats.pt --target $g --target-stats source --splits-json data/$g/splits/audit_all.json \
    --n-test 0 --no-audit --device cpu --out $OUT/${size}_seed0_$g.json
echo K_AUDIT_DONE
