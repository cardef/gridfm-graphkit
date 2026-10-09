#!/bin/bash
# Fetch a spread of partitions (0 and every 50th: 20 of ~1000, about 4,400 scenarios) of the public PF datasets of
# case500_goc and case2000_goc, bus/branch/gen tables only, for the size-extrapolation evaluation (P47-P50). The full
# sets are 13 and 80 GB and the filesystem is 98% full. Login node only (compute nodes have no network). Idempotent.
#   nohup bash fetch_extrap.sh > logs/fetch_extrap.log 2>&1 &
set -uo pipefail
R=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/.hf-home
HF=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/HierarchicalFM/upstream/.hfvenv/bin/hf

retry() {
    for i in 1 2 3 4 5 6; do
        "$@" && return 0
        echo "RETRY $i: ${*: -3}"
        sleep 60
    done
    return 1
}

inc=()
for t in bus_data branch_data gen_data; do
    inc+=(--include "$t.parquet/scenario_partition=0/*" --include "$t.parquet/scenario_partition=*00/*" \
          --include "$t.parquet/scenario_partition=*50/*")
done
for c in case500_goc case2000_goc; do
    retry "$HF" download "gridfm/pf_small_$c" --repo-type dataset --local-dir "$R/data/genco/full/$c/raw" "${inc[@]}" \
        --include n_scenarios.txt --include args.log --include README.md --format quiet >/dev/null &&
        echo "DATA_DONE $c $(date -Is) $(du -sh "$R/data/genco/full/$c/raw" | cut -f1)"
done
echo "FETCH_DONE $(date -Is)"
