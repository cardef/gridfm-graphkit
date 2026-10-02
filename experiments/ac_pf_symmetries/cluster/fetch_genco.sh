#!/bin/bash
# Fetch the released GENCO PF checkpoints for IEEE 14/30/57/118 (every size and seed, with their MLflow eval
# artifacts: official split ids per seed, logged test metrics) and the full public PF datasets (bus, branch
# and gen tables only: neither preprocessing nor the normalizer reads the others).
# Login node only (compute nodes have no network). Idempotent: hf skips files already present.
#   nohup bash fetch_genco.sh > logs/fetch_genco.log 2>&1 &
set -uo pipefail
R=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/.hf-home
HF=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/HierarchicalFM/upstream/.hfvenv/bin/hf
CASES=(case14_ieee case30_ieee case57_ieee case118_ieee)

retry() {
    for i in 1 2 3 4 5 6; do
        "$@" && return 0
        echo "RETRY $i: ${*: -3}"
        sleep 60
    done
    return 1
}

inc=()
for c in "${CASES[@]}"; do inc+=(--include "$c/*" --include "mlflow/eval/$c/*"); done
retry "$HF" download gridfm/genco-pf-datakit "${inc[@]}" --local-dir "$R/data/genco_ckpt" --format quiet >/dev/null &&
    echo "CKPT_DONE $(date -Is)"

for c in "${CASES[@]}"; do
    retry "$HF" download "gridfm/pf_small_$c" --repo-type dataset --local-dir "$R/data/genco/full/$c/raw" \
        --include "bus_data.parquet/*" --include "branch_data.parquet/*" --include "gen_data.parquet/*" \
        --include n_scenarios.txt --include args.log --include README.md --format quiet >/dev/null &&
        echo "DATA_DONE $c $(date -Is)"
done
echo "FETCH_DONE $(date -Is)"
