#!/bin/bash
#SBATCH --job-name=symfm-data
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%j.out
# Training data of the cluster replicate: 2048 random scenarios (seed 0) of each of HierarchicalFM's
# 10k-scenario datakit sets (base topology, agg_load_profile; generated 2026-07-08), the PoC's data size.
# Processed once here, so the training array never races on the cache.
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
CASES=(case14_ieee case30_ieee case57_ieee)
for c in "${CASES[@]}"; do
    [ -d "data/$c/raw" ] || python -m experiments.ac_pf_symmetries.subset_data random \
        --src "$S/HierarchicalFM/gridfm-graphkit/data/$c/raw" --out "data/$c" --n 2048 --seed 0
done
python - "${CASES[@]/#/data/}" <<'EOF'
import sys

import yaml

from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.param_handler import NestedNamespace, load_normalizer

args = NestedNamespace(**yaml.safe_load(open("examples/config/HGNS_PF_datakit_case14.yaml")))
for root in sys.argv[1:]:
    print(root, len(HeteroGridDatasetDisk(root, load_normalizer(args))), flush=True)
EOF
echo DATA_DONE
