#!/bin/bash
# R001 full local-scale regeneration (disk guard: min 5GB free)
cd /Users/carmine/Code/FM/gridfm-graphkit
for g in case14_ieee case30_ieee case57_ieee case118_ieee case500_goc case2000_goc Texas2k_case1_2016summerpeak; do
  free_kb=$(df -k /System/Volumes/Data | tail -1 | awk '{print $4}')
  if [ "$free_kb" -lt 5242880 ]; then echo "ABORT before $g: ${free_kb}KB free"; exit 2; fi
  echo "=== $g start $(date +%H:%M:%S) (free: ${free_kb}KB) ==="
  ../.venv/bin/gridfm_datakit generate experiments/m0/datakit_configs/$g.yaml > experiments/m0/logs/gen_$g.log 2>&1 || { echo "FAIL $g"; exit 1; }
  echo "OK $g $(date +%H:%M:%S) ($(du -sh data/$g 2>/dev/null | cut -f1))"
done
echo ALL_DONE
