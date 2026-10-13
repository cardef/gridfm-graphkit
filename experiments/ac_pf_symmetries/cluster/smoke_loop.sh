#!/bin/bash
# Smoke test of the closed-loop arm (dcloopcanon) on CPU: case14 and case30 as the training list, 64 scenarios each,
# 2 epochs, held out case57, then --eval-only --extrap case118. Small grids only: MKL's batched float32 solve hangs on
# 500-bus graphs on CPU. Prints the result lines and the per-layer validation residuals of the last epoch.
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
srun --partition=vicomtech --account=di13 --qos=qos_di13 --nodelist=abacus-004 --cpus-per-task=4 --mem=16G --time=00:40:00 \
    --job-name=symfm-loopsmoke bash -c "source $S/env/activate.sh $R && cd $R && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
    && printf 'case14_ieee\ncase30_ieee\n' > /tmp/loop_smoke_list.txt && O=$S/SymmetricalFM/scratch/loopsmoke && rm -rf \$O \
    && export SYMFM_DATA_ROOT=$R/data \
    && python -m experiments.ac_pf_symmetries.multigrid --arm dcloopcanon --seeds 0 --train-file /tmp/loop_smoke_list.txt \
       --heldout case57_ieee --n-per-grid 64 --n-heldout 32 --epochs 2 --workers 2 --zs-refit --device cpu --results-dir \$O \
       2>&1 | grep -E '^\{|Error|Traceback' \
    ; python -m experiments.ac_pf_symmetries.multigrid --eval-only --extrap case118_ieee --arm dcloopcanon --seeds 0 \
       --train-file /tmp/loop_smoke_list.txt --heldout case57_ieee --n-per-grid 64 --n-heldout 32 --zs-refit --device cpu \
       --results-dir \$O 2>&1 | grep -E '^\{|Error|Traceback' \
    ; python - \$O <<'PY'
import csv, glob, sys
for f in glob.glob(sys.argv[1] + '/*/*/logs/version_0/metrics.csv'):
    rows = [r for r in csv.DictReader(open(f)) if r['Validation loss']]
    for r in rows:
        print('epoch', r['epoch'], 'val', r['Validation loss'], 'layer residuals',
              ' '.join('%.3g' % float(r[f'Validation layer_{k}_residual']) for k in range(12)))
PY
    echo SMOKE_EXIT"
