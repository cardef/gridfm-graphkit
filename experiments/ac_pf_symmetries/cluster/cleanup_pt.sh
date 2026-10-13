#!/bin/bash
# Remove the X3 node-local caches (/scratch/symfm_cache_x3) from the GPU nodes that ran train_pt.sbatch.
#   bash cleanup_pt.sh abacus-002 abacus-003 abacus-006
for n in "$@"; do
    case $n in abacus-00[23]) g=a40 ;; *) g=h100 ;; esac
    srun --partition=vicomtech --account=di13 --qos=qos_di13 --nodelist=$n --gres=gpu:$g:1 --cpus-per-task=1 --mem=1G \
        --time=00:10:00 --job-name=symfm-clean bash -c "du -sh /scratch/symfm_cache_x3 2>/dev/null; rm -rf /scratch/symfm_cache_x3 /scratch/symfm_cache_x3.lock; echo CLEANED $n"
done
