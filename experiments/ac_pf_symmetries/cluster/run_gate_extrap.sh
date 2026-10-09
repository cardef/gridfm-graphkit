#!/bin/bash
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
srun --partition=vicomtech --account=di13 --qos=qos_di13 --cpus-per-task=4 --mem=16G --time=01:00:00 --job-name=symfm-gate \
    bash -c "source $S/env/activate.sh $R && cd $R && export OMP_NUM_THREADS=4 && python experiments/ac_pf_symmetries/gate_extrap.py case2000_goc > $S/SymmetricalFM/scratch/gate_2000.txt 2> $S/SymmetricalFM/scratch/gate_2000.err; echo GATE_DONE \$?"
