#!/bin/bash
# Piano X, X0 (frontier.py): smoke on 8 scenarios of case14, then the six grids; summary in results/pianox/FRONTIER.txt.
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
srun --partition=vicomtech --account=di13 --qos=qos_di13 --cpus-per-task=4 --mem=24G --time=03:00:00 --job-name=symfm-x0 \
    bash -c "source $S/env/activate.sh $R && cd $R && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 && echo HEAD \$(git rev-parse HEAD) && bash experiments/ac_pf_symmetries/cluster/provenance.sh experiments/ac_pf_symmetries/{frontier,r2_classical,dcprior,multigrid,run}.py && mkdir -p experiments/ac_pf_symmetries/results/pianox && python -m experiments.ac_pf_symmetries.frontier case14_ieee --n 8 --out $S/SymmetricalFM/scratch/x0smoke && echo SMOKE_OK && python -m experiments.ac_pf_symmetries.frontier case14_ieee case30_ieee case57_ieee case118_ieee case500_goc case2000_goc > experiments/ac_pf_symmetries/results/pianox/FRONTIER.txt && echo X0_DONE"
