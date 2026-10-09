#!/bin/bash
# Smoke test of the Piano K arms (plain, gp) and of --split: 1 epoch on 64 scenarios of case14 (srun, minutes).
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
srun --partition=vicomtech --account=di13 --qos=qos_di13 --cpus-per-task=4 --mem=16G --time=00:40:00 --job-name=symfm-ksmoke \
    bash -c "source $S/env/activate.sh $R && cd $R && export OMP_NUM_THREADS=4 && python - <<'PY'
import json, random
n = 2048
ids = list(range(n)); random.Random(0).shuffle(ids)
json.dump({'train': ids[:48], 'val': ids[48:56], 'test': ids[56:64]}, open('/tmp/ksmoke_split.json', 'w'))
PY
python -m experiments.ac_pf_symmetries.multigrid --arm plain gp dccanon --seeds 0 --train case14_ieee --heldout case30_ieee \
    --split case14_ieee=/tmp/ksmoke_split.json --n-per-grid 64 --epochs 1 --device cpu \
    --results-dir $S/SymmetricalFM/scratch/ksmoke 2>&1 | tail -12; echo SMOKE_EXIT \${PIPESTATUS[0]}"
