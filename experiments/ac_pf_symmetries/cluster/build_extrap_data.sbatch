#!/bin/bash
#SBATCH --job-name=symfm-xdata
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%j.out
# Size-extrapolation data (P47-P50): 2048 random scenarios (seed 0) of the fetched partitions of case500_goc and
# case2000_goc (fetch_extrap.sh), processed once so the evaluation array never races on the cache; then the
# references on their fixed 1024-scenario subset (r2_baselines.py, r2_classical.py), computed before any model is
# evaluated there. Needs fetch_extrap.sh to have finished.
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
CASES=(case500_goc case2000_goc)
for c in "${CASES[@]}"; do
    grep -q "DATA_DONE $c" experiments/ac_pf_symmetries/cluster/logs/fetch_extrap.log || { echo "fetch of $c not done"; exit 1; }
    [ -d "data/$c/raw" ] || python -m experiments.ac_pf_symmetries.subset_data random \
        --src "data/genco/full/$c/raw" --out "data/$c" --n 2048 --seed 0
done
python - "${CASES[@]/#/data/}" <<'PY'
import sys

import yaml

from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.param_handler import NestedNamespace, load_normalizer

args = NestedNamespace(**yaml.safe_load(open("examples/config/HGNS_PF_datakit_case14.yaml")))
for root in sys.argv[1:]:
    print(root, len(HeteroGridDatasetDisk(root, load_normalizer(args))), flush=True)
PY
echo "HEAD $(git rev-parse HEAD) host=$(hostname)"
bash experiments/ac_pf_symmetries/cluster/provenance.sh \
    experiments/ac_pf_symmetries/{r2_baselines,r2_classical,multigrid,hodge,dcprior,run}.py
OUT=experiments/ac_pf_symmetries/results/abacus_r2
python -m experiments.ac_pf_symmetries.r2_baselines "${CASES[@]}" > $OUT/BASELINES_extrap.txt
python -m experiments.ac_pf_symmetries.r2_classical "${CASES[@]}" > $OUT/CLASSICAL_extrap.txt
du -sh data/case500_goc data/case2000_goc
echo XDATA_DONE
