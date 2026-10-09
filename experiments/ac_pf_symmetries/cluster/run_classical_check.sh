#!/bin/bash
# Validate the sparse r2_classical.py against the dense outputs (CLASSICAL.txt case14, CLASSICAL_extrap.txt case500),
# then run it on case2000_goc; output in scratch/classical_sparse_*.txt via the caller.
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
srun --partition=vicomtech --account=di13 --qos=qos_di13 --cpus-per-task=4 --mem=16G --time=01:30:00 --job-name=symfm-sparse \
    bash -c "source $S/env/activate.sh $R && cd $R && export OMP_NUM_THREADS=4 && echo HEAD \$(git rev-parse HEAD) && sha1sum experiments/ac_pf_symmetries/r2_classical.py && (time python -m experiments.ac_pf_symmetries.r2_classical case14_ieee case500_goc) > $S/SymmetricalFM/scratch/classical_sparse_check.txt 2> $S/SymmetricalFM/scratch/classical_sparse_check.err && echo CHECK_DONE && (time python -m experiments.ac_pf_symmetries.r2_classical case2000_goc) > $S/SymmetricalFM/scratch/classical_sparse_2000.txt 2> $S/SymmetricalFM/scratch/classical_sparse_2000.err && echo SPARSE_2000_DONE"
