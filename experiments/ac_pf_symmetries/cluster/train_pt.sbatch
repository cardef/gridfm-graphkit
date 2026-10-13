#!/bin/bash
#SBATCH --job-name=symfm-pt
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%j.out
# Piano X, X3 (GPU): train on the 31 public subgrids (cluster/pt_train_networks.txt, up to 1,200 scenarios each) and
# evaluate zero-shot on the fixed 1024-scenario sets of case118_ieee (at the end of training), then case118_tr,
# case500_goc, case2000_goc, case57_ieee, case30_ieee, case14_ieee (--eval-only --extrap), with the source and the
# target-fitted normalizer. Raw data copied to the job's private node-local /tmp (Slurm's per-job tmpfs on the local
# disk, removed at the end of the job) and processed there: per-scenario caches do not fit on GPFS.
# One run per GPU (the QoS allows 2 GPUs per job), RUNS="arm:seed ..." in the order of the allocated GPUs.
# Needs prep_pt.sbatch to have finished.
#   sbatch --nodelist=abacus-002 --export=ALL,RUNS="plain:0 dcinjcanon:0" train_pt.sbatch
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LIST=experiments/ac_pf_symmetries/cluster/pt_train_networks.txt
LOGD=experiments/ac_pf_symmetries/cluster/logs
OUT=$R/experiments/ac_pf_symmetries/results/pianox_pt
TEST="case118_ieee case118_tr case500_goc case2000_goc case57_ieee case30_ieee case14_ieee"
[ -s "$LIST" ] || { echo "no $LIST: run prep_pt.sbatch first"; exit 1; }
L=$TMPDIR/symfm_cache_x3
echo "HEAD $(git rev-parse HEAD) host=$(hostname) RUNS=$RUNS $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
bash experiments/ac_pf_symmetries/cluster/provenance.sh \
    experiments/ac_pf_symmetries/{multigrid,hodge,run,symmetries,dcprior,resnorm}.py
(
flock 9
if [ ! -f $L/READY ]; then
rm -rf $L
mkdir -p $L
for g in $(cat $LIST) $TEST; do mkdir -p $L/$g && cp -r data/$g/raw $L/$g/; done
python - $L $(cat $LIST) $TEST <<'PY'
import sys
from multiprocessing import Pool


def build(root):
    import yaml

    from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
    from gridfm_graphkit.io.param_handler import NestedNamespace, load_normalizer

    args = NestedNamespace(**yaml.safe_load(open("examples/config/HGNS_PF_datakit_case14.yaml")))
    return root, len(HeteroGridDatasetDisk(root, load_normalizer(args)))


if __name__ == "__main__":
    with Pool(10) as pool:
        for r, n in pool.imap_unordered(build, [f"{sys.argv[1]}/{g}" for g in sys.argv[2:]]):
            print(r, n, flush=True)
PY
touch $L/READY
echo "DATA_BUILT $(date -Is) $(du -sh $L | cut -f1)"
fi
) 9>$TMPDIR/symfm_cache_x3.lock
[ -f $L/READY ] || { echo "cache not ready"; exit 1; }
export SYMFM_DATA_ROOT=$L
echo "DATA_READY $(date -Is)"
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
common="--train-file $LIST --heldout case118_ieee --n-per-grid 1200 --n-heldout 1024 --device cuda --zs-refit --results-dir $OUT"
i=0
for r in $RUNS; do
    arm=${r%:*}
    seed=${r#*:}
    dev=${GPUS[$i]}
    (CUDA_VISIBLE_DEVICES=$dev python -m experiments.ac_pf_symmetries.multigrid --arm $arm --seeds $seed --workers 4 $common \
        > $LOGD/pt_${arm}_s${seed}.log 2>&1 \
     && CUDA_VISIBLE_DEVICES=$dev python -m experiments.ac_pf_symmetries.multigrid --eval-only \
        --extrap case118_tr case500_goc case2000_goc case57_ieee case30_ieee case14_ieee --arm $arm --seeds $seed $common \
        >> $LOGD/pt_${arm}_s${seed}.log 2>&1 \
     && echo "RUN_DONE $arm $seed $(date -Is)" || echo "RUN_FAILED $arm $seed $(date -Is)") &
    i=$((i + 1))
done
wait
echo PT_DONE
