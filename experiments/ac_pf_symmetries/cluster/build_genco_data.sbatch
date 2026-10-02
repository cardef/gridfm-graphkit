#!/bin/bash
#SBATCH --job-name=symfm-gdata
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%A_%a.out
# Per grid (array task = grid): E0p on the full public dataset, then the reduced copy holding the scenarios
# of the released runs' official splits (every seed's test split, first 5000 train / 500 val ids), processed
# once so the audit array never races on the cache. Needs fetch_genco.sh to have finished.
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
CASES=(case14_ieee case30_ieee case57_ieee case118_ieee)
c=${CASES[$SLURM_ARRAY_TASK_ID]}
G=data/genco
grep -q "DATA_DONE $c" experiments/ac_pf_symmetries/cluster/logs/fetch_genco.log || { echo "fetch of $c not done"; exit 1; }
python -m experiments.ac_pf_symmetries.data_audit --root $G/full --cases "$c" --max-partitions 100000 \
    --out "experiments/ac_pf_symmetries/results/genco/e0p_$c.json"
[ -d "$G/r/$c/raw" ] || python -m experiments.ac_pf_symmetries.subset_data official --src "$G/full/$c/raw" \
    --runs "data/genco_ckpt/mlflow/eval/$c" --out "$G/r/$c" --n-train 5000 --n-val 500
python - "$G/r/$c" <<'EOF'
import sys

import yaml

from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.param_handler import NestedNamespace, load_normalizer

args = NestedNamespace(**yaml.safe_load(open("examples/config/HGNS_PF_datakit_case14.yaml")))
print(sys.argv[1], len(HeteroGridDatasetDisk(sys.argv[1], load_normalizer(args))), flush=True)
EOF
echo "GDATA_DONE $c"
