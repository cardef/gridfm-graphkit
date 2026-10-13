#!/bin/bash
# Piano X, X3 data: (1) the public pretrained GENCO checkpoint, its config, the case118 evaluation set of the transfer
# experiment (9,952 scenarios) and the 100- and 1,000-sample fine-tuned checkpoints (gridfm/genco-pf-transfer);
# (2) a spread of scenario partitions (0, 8, 16, 24, 32, 40 where present: 4-6 of 27-50, about 1,000-1,600 scenarios
# per grid) of the 31 pretraining subgrids that are public (gridfm/genco-pf-pretraining lists 100 training, 10
# validation and 10 test subgrids; only 31 training ones are uploaded as of 2026-10-13), bus/branch/gen tables only.
# Login node only. Idempotent.  nohup bash fetch_pretrain.sh > logs/fetch_pretrain.log 2>&1 &
set -uo pipefail
R=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/.hf-home
HF=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/HierarchicalFM/upstream/.hfvenv/bin/hf
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "RETRY $i"; sleep 60; done; return 1; }
retry "$HF" download gridfm/genco-pf-transfer --include "pretraining/*" --include "data/case118_ieee/raw/*" \
    --include "finetune/case118_100/*" --include "finetune/case118_1000/*" --include "README.md" \
    --local-dir "$R/data/genco_transfer" --format quiet >/dev/null && echo "TRANSFER_DONE $(date -Is) $(du -sh $R/data/genco_transfer | cut -f1)"
inc=()
for t in bus_data branch_data gen_data; do
    for p in 0 8 16 24 32 40; do inc+=(--include "*/$t.parquet/scenario_partition=$p/*"); done
done
retry "$HF" download gridfm/genco-pf-pretraining --repo-type dataset "${inc[@]}" --include "*/n_scenarios.txt" --include "*/args.log" \
    --include "*.txt" --include "README.md" --local-dir "$R/data/genco_pretrain" --format quiet >/dev/null && \
    echo "PRETRAIN_DONE $(date -Is) $(du -sh $R/data/genco_pretrain | cut -f1) $(ls -d $R/data/genco_pretrain/subnet_* | wc -l) grids"
echo "FETCH_DONE $(date -Is)"
