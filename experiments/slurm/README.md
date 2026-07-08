# M1 SLURM launch scripts

Runs the cluster stage of the M1 pack (`experiments/m1/README.md`): datagen
-> Kron-Schur precompute/gate -> R010 training matrix -> R011 memory sweep.

No SLURM cluster is documented anywhere in this repo or its history — the
only cluster in `README.md` is LSF (`bsub`). Everything cluster-specific
here is a placeholder; fill in `cluster.env` first.

## Before you submit

1. **`cluster.env`**: fill in every `CHANGE_ME`. `VENV_ACTIVATE` must point
   at an environment that already has this repo installed editable
   (`uv pip install -e ".[dev,test]"`) with `gridfm-datakit` as an editable
   sibling checkout, per the repo's `CLAUDE.md` — these scripts activate an
   environment, they don't build one.
2. **`JULIA_DEPOT_PATH`**: point it at shared, persistent storage, not
   node-local `/tmp`. A local dev session already hit "PythonCall.jl did not
   start properly" after `/tmp` got cleaned. On a fresh depot, warm it with
   one array task before launching all 7 in parallel (comment in
   `01_datagen.sbatch`) — concurrent first-time Julia precompilation into an
   empty shared depot can race.
3. **`--time` / `--mem` / `--cpus-per-task`** in each `.sbatch` file are
   unverified placeholders, not measurements — the only local timing data is
   at M0 scale (512–2048 scenarios/grid); M1 requests 10k–40960. Watch the
   first datagen array (especially `case2000_goc`, 40960 scenarios
   requested) and adjust `--time` before relying on it unattended.
4. **`--array=1-18%6`** in `03_train_r010.sbatch` is a fallback for
   standalone submission. Normal path: `02_precompute_gate.sbatch` submits
   03 itself once it knows the real config count (see below) — if your
   cluster disallows `sbatch` from inside a running job, submit 03 by hand
   after 02 finishes: `sbatch --array=1-$(ls experiments/m1/configs/*.yaml | wc -l)%6 experiments/slurm/03_train_r010.sbatch`.

## Submit

```bash
experiments/slurm/submit_all.sh
```

Or step by step: `sbatch experiments/slurm/01_datagen.sbatch`, then after it
finishes `sbatch experiments/slurm/02_precompute_gate.sbatch` (which chains
03 itself), and independently `sbatch experiments/slurm/04_train_r011_phaseb.sbatch`.

## Known gaps — read before trusting results

- **R007 scenario filter is not implemented.** `r007_outlier_triage.py`
  only diagnoses the case2000 dead-bus artifact (verdict: "flag scenarios
  where a zero-load bus has |V|<0.1 p.u."); no code in this repo actually
  drops those scenarios anywhere in the data or training pipeline. `02` reruns
  the diagnostic against the cluster-scale data and stops there — implement
  and wire in the filter yourself before trusting case2000 M2-gate numbers.
- **Seed asymmetry.** The M1 plan wants 3 seeds at case500, 2 at case2000;
  `r010_make_configs.py --seeds 0 1 2` (run inside `02`) applies the same
  seed list to both grids (`M1_GRIDS` loop, no per-grid override), so it
  emits a case2000 seed-2 config the plan didn't ask for. Harmless
  over-generation, not a silent under-count — skip or delete that config if
  you want to hold to the original plan exactly.
- **No multi-node DDP wiring for SLURM.** `gridfm_graphkit/__main__.py`
  auto-configures `MASTER_ADDR`/`NODE_RANK`/NCCL only when it detects LSF
  env vars (`is_lsf()`); there is no equivalent SLURM path. Irrelevant for
  this pipeline (every R010/R011 job is single-GPU — batch sizes 4–8, one
  `--gres=gpu:1` each), but don't point `--array` tasks at multi-GPU nodes
  expecting DDP to Just Work.
- **`--compile` mode per grid (case500 -> reduce-overhead, case2000 ->
  max-autotune) is an inferred reading** of `experiments/m1/README.md`'s E4
  note, which states the case2000/`<=case118` endpoints but not case500
  explicitly. Confirm before trusting `torch.compile` behavior differs from
  what you intended.
