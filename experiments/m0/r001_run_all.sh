#!/usr/bin/env bash
# R001 full local-scale regeneration (disk guard: min 5GB free)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATAKIT_BIN="${GRIDFM_DATAKIT_BIN:-}"
if [[ -n "$DATAKIT_BIN" ]]; then
  DATAKIT_BIN="$(command -v "$DATAKIT_BIN" || true)"
elif command -v gridfm_datakit >/dev/null 2>&1; then
  DATAKIT_BIN="$(command -v gridfm_datakit)"
elif [[ -x "$REPO_ROOT/../.venv/bin/gridfm_datakit" ]]; then
  DATAKIT_BIN="$REPO_ROOT/../.venv/bin/gridfm_datakit"
fi
if [[ -z "$DATAKIT_BIN" ]]; then
  echo "gridfm_datakit not found; activate an environment or set GRIDFM_DATAKIT_BIN" >&2
  exit 127
fi

mkdir -p "$SCRIPT_DIR/logs"
cd "$REPO_ROOT"
for g in case14_ieee case30_ieee case57_ieee case118_ieee case500_goc case2000_goc Texas2k_case1_2016summerpeak; do
  free_kb=$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')
  if [ "$free_kb" -lt 5242880 ]; then echo "ABORT before $g: ${free_kb}KB free"; exit 2; fi
  echo "=== $g start $(date +%H:%M:%S) (free: ${free_kb}KB) ==="
  "$DATAKIT_BIN" generate "experiments/m0/datakit_configs/$g.yaml" > "experiments/m0/logs/gen_$g.log" 2>&1 || { echo "FAIL $g"; exit 1; }
  echo "OK $g $(date +%H:%M:%S) ($(du -sh data/$g 2>/dev/null | cut -f1))"
done
echo ALL_DONE
