#!/bin/bash
#SBATCH --job-name=symfm-genco
#SBATCH --partition=vicomtech
#SBATCH --account=di13
#SBATCH --qos=qos_di13
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit/experiments/ac_pf_symmetries/cluster/logs/%x-%A_%a.out
# F0 of Piano A on one released GENCO PF checkpoint per array task (36 = 4 grids x base/small/tiny x seeds
# 0/1/42; task i -> CASES[i/9], SIZES[(i/3)%3], SEEDS[i%3]), data = the reduced official-split copies:
#   G1-G3  full official test split, accuracy only (strict load, split size, stats applied: asserted)
#   gate   the framework's refit on the 5000-id train subset, compared offline with the released baseMVA
#   E1/E2  audit raw / post-hoc Canonicalize on the first 2000 official test graphs
#   E5     every other grid (its base seed-0 test split, same 2000 graphs for every checkpoint): target
#          refit; source-fixed normalizer + audit + the S3 probe at k = refit/source base (the refit's
#          exact action); canon with this checkpoint's scale_ref under both normalizers
# MODE=gates stops after the gates. Outputs already present are skipped (requeue-safe).
#   sbatch --array=0-35%5 genco_audit.sbatch    (CPU: ~150 graphs/s on 8 cores, ~2x below a T4 here;
#   the L4 node is not open to qos_di13; add --gres=gpu:t4:1 to run on a T4)
set -euo pipefail
S=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO
R=$S/SymmetricalFM/gridfm-graphkit
source $S/env/activate.sh $R
cd $R
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
CASES=(case14_ieee case30_ieee case57_ieee case118_ieee)
SIZES=(base small tiny)
SEEDS=(0 1 42)
i=$SLURM_ARRAY_TASK_ID
c=${CASES[$((i / 9))]}
z=${SIZES[$(((i / 3) % 3))]}
s=${SEEDS[$((i % 3))]}
tag=${c}_${z}_s$s
D=data/genco_ckpt/$c/$z/seed$s
SPL=data/genco/r/$c/splits/${z}_seed$s.json
O=experiments/ac_pf_symmetries/results/genco
echo "HEAD $(git rev-parse HEAD) $tag host=$(hostname)"
bash experiments/ac_pf_symmetries/cluster/provenance.sh experiments/ac_pf_symmetries/{genco_audit,run,symmetries}.py
nvidia-smi -L 2>/dev/null || echo "CPU x$SLURM_CPUS_PER_TASK"
A=(-m experiments.ac_pf_symmetries.genco_audit --config "$D/${c}_${z}_seed$s.yaml" --data-path data/genco/r
    --model-path "$D/best_model_state_dict.pt" --normalizer-stats "$D/normalizer_stats.pt")
run() {
    local out=$1
    shift
    if [ -f "$out" ]; then echo "skip $out"; return; fi
    echo "=== $out $*"
    python "${A[@]}" "$@" --out "$out.tmp" && mv "$out.tmp" "$out"
}
run "$O/g1/$tag.json" --no-audit --n-test 0 --splits-json "$SPL"
run "$O/gate_refit/$tag.json" --no-audit --n-test 200 --splits-json "$SPL" --target "$c" --target-stats refit
[ "${MODE:-all}" = gates ] && { echo GATES_DONE; exit 0; }
run "$O/e1/$tag.json" --n-test 2000 --splits-json "$SPL"
run "$O/e2/$tag.json" --canon --n-test 2000 --splits-json "$SPL"
sref=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['scale_ref'])" "$O/e2/$tag.json")
for t in "${CASES[@]}"; do
    [ "$t" = "$c" ] && continue
    TS=data/genco/r/$t/splits/base_seed0.json
    E=$O/e5/${tag}__${t}
    run "${E}_refit.json" --no-audit --n-test 2000 --splits-json "$TS" --target "$t" --target-stats refit
    # the refit normalizer is exactly S3 with k = baseMVA_refit / baseMVA_source: probe that k
    k=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['baseMVA'] / json.load(open(sys.argv[2]))['baseMVA'])" \
        "${E}_refit.json" "$O/e1/$tag.json")
    run "${E}_source.json" --n-test 2000 --splits-json "$TS" --target "$t" --target-stats source --scale-probe "$k"
    for st in source refit; do
        run "${E}_canon_$st.json" --no-audit --canon --scale-ref "$sref" --n-test 2000 --splits-json "$TS" \
            --target "$t" --target-stats "$st"
    done
done
echo AUDIT_DONE
