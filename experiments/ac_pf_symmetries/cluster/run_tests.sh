#!/bin/bash
# Run the experiment test suite on a CPU node (srun); output in logs/tests_<tag>.log via the caller.
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
srun --partition=vicomtech --account=di13 --qos=qos_di13 --cpus-per-task=4 --mem=16G --time=00:45:00 --job-name=symfm-tests \
    bash -c "source $S/env/activate.sh $R && cd $R && git rev-parse HEAD && git status --short | head -20 && python -m pytest -q experiments/ac_pf_symmetries/tests 2>&1 | tail -20; echo TESTS_EXIT \${PIPESTATUS[0]}"
