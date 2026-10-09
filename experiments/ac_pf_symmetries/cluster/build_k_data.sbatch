#!/bin/bash
#SBATCH --job-name=symfm-kdata
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%j.out
# Piano K data: (1) data/case118_k = the official seed-0 split of the public N-1 case118 set (5000 train, 500 val,
# first 2000 test) from the reduced copy data/genco/r/case118_ieee, with splits/official_seed0.json; (2) processed
# caches of case118_k and of the N-k sets data/case118_nk<k> (gen_nk.sbatch), built once so the arrays never race;
# (3) audit split files for the released checkpoints on every N-k set (10 train / 10 val / rest test);
# (4) DC/flat and classical references on every N-k set (the fixed 1024 of each), before any model is evaluated there.
#   sbatch --dependency=afterok:<gen_nk array id> build_k_data.sbatch
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "HEAD $(git rev-parse HEAD) host=$(hostname) $(date -Is)"
[ -d data/case118_k/raw ] || python -m experiments.ac_pf_symmetries.subset_data ids \
    --src data/genco/r/case118_ieee/raw --split data/genco/r/case118_ieee/splits/base_seed0.json \
    --n-test 2000 --name official_seed0 --out data/case118_k
NK=(case118_nk1 case118_nk2 case118_nk3 case118_nk5 case118_nk10 case118_nk20)
for g in "${NK[@]}"; do [ -d "data/$g/raw" ] || { echo "data/$g/raw missing: run gen_nk.sbatch first"; exit 1; }; done
python - data/case118_k "${NK[@]/#/data/}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.param_handler import NestedNamespace, load_normalizer

args = NestedNamespace(**yaml.safe_load(open("examples/config/HGNS_PF_datakit_case14.yaml")))
for root in sys.argv[1:]:
    n = len(HeteroGridDatasetDisk(root, load_normalizer(args)))
    print(root, n, flush=True)
    if "_nk" in root:  # audit split for the released checkpoints: everything but 20 scenarios is the test set
        d = Path(root) / "splits"
        d.mkdir(exist_ok=True)
        (d / "audit_all.json").write_text(json.dumps({"train": list(range(10)), "val": list(range(10, 20)), "test": list(range(20, n))}))
PY
bash experiments/ac_pf_symmetries/cluster/provenance.sh \
    experiments/ac_pf_symmetries/{r2_baselines,r2_classical,multigrid,hodge,dcprior,run,subset_data}.py
OUT=experiments/ac_pf_symmetries/results/pianok
mkdir -p $OUT
python -m experiments.ac_pf_symmetries.r2_baselines "${NK[@]}" > $OUT/BASELINES_nk.txt
python -m experiments.ac_pf_symmetries.r2_classical "${NK[@]}" > $OUT/CLASSICAL_nk.txt
du -sh data/case118_k data/case118_nk*
echo KDATA_DONE
