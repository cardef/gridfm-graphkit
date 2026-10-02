# AC power-flow symmetries: equivariance audit PoC

Proof of concept for [Piano A](../../AC_PF_SYMMETRIES_PLAN.md) (F0/F1): catalog the
exact symmetries of the AC-PF map, measure how much a GridFM PF surrogate breaks
them (equivariance error `EE_g`), and show that an exact fix (canonicalization,
Piano A Prop. 5) costs nothing in-distribution while augmentation only covers the
range it was sampled from.

From the repository root, in an environment with `gridfm_graphkit` installed
(`.venv-screening` on the development machine):

```bash
.venv-screening/bin/python -m pytest -q experiments/ac_pf_symmetries/tests
.venv-screening/bin/python -m experiments.ac_pf_symmetries.run --variant m0 aug augcov canon --seeds 0 1 2
.venv-screening/bin/python -m experiments.ac_pf_symmetries.report --case case14_ieee
# re-evaluate saved models with the current code (writes result_eval.json, never result.json)
.venv-screening/bin/python -m experiments.ac_pf_symmetries.run --eval-only --variant m0 aug augmild augphase augscale canon
.venv-screening/bin/python -m experiments.ac_pf_symmetries.report --case case14_ieee --result-file result_eval.json
```

## Symmetries implemented (`symmetries.py`)

Actions act on the normalized, masked hetero batch the model sees, and on the
output layout `[VM, VA, PG, QG]`.

| id | action on inputs | expected action on outputs |
|---|---|---|
| S1 `phase` | `VA_ref += alpha`, labels `VA += alpha` | `VA += alpha` |
| S3 `scale` | every power/admittance column `/= k` (P, Q, Q limits, GS, BS, Yff, Yft, rate), V untouched | `PG, QG /= k`; `VM, VA` invariant |
| S4 `flip` | re-declare from/to of a random 50% of the lines: swap their forward/reverse rows | identity |
| S4 `fliptrafo` | re-declare from/to of every transformer: swap the rows, `tap -> 1/tap`, angle limits negated and swapped | identity |

Each directed edge row stores its own `(flow, Y_self, Y_mutual)`, so the row swap
already carries the MATPOWER reversal `(tau, phi, z, b) -> (1/tau, -phi, tau^2 z,
b/tau^2)`. Only `tap` and the angle limits are stored in the from->to convention
on both rows. For lines the flip is a pure row permutation and exact by
construction. For transformers `tap` is the one orientation-dependent input the
model sees (`ang_min/ang_max` are masked). In AC the flip swaps the two branch
flows; it does not negate them, because their sum is the branch loss.

S2 (local gauge) would act on any grid by creating virtual phase shifts on lines.
On data without phase shifters its constraint reduces to S1 (Piano A, Prop. 6),
and none of the local datakit cases has one (`shift == 0`, `Yft == Ytf`), so it is
not implemented. S5 (node permutation) holds for any GNN and is not measured.

`EE_g` is reported per channel, in the channel's unit (VA wrapped to `[-pi,
pi)`), and relative to the model's own in-distribution RMSE on that channel
(`report.py`): above 1, the symmetry breaking dominates the error on `T_g`.

`tests/test_symmetries.py` verifies on the repository fixture that:

1. the labelled states are PF solutions in the framework's own physics;
2. the power-balance residual of a *perturbed* state transforms exactly under each action (checked on O(0.1) numbers, not on solver noise);
3. the transformer flip equals the MATPOWER reversed device, and the naive rule without `tau^2` does not;
4. the output action matches the label action;
5. a bare `GNS_heterogeneous` breaks S1, S3 and S4 on transformers but not S4 on lines;
6. `Canonicalize` is equivariant to float precision;
7. in `CovariantAugment` the rescaled physics residual equals the physics of the de-augmented prediction on the original sample.

## Models (`run.py`)

* `m0`: `GNS_heterogeneous` with the official `HGNS_PF_datakit_case14.yaml`
  hyper-parameters (12 layers, hidden 48, 8 heads, physics + masked-MSE loss).
* `aug`: same, trained with random S1 (`alpha ~ U(-pi, pi)`) and S3
  (`k ~ logU(0.1, 10)`) actions applied per sample. This is the "wide" setting,
  which covers the audited range. The loss sees the augmented sample, so the
  weight of the power residuals and of the PG/QG errors drifts like `1/k`: a
  handicapped baseline, kept as a documented failure.
* `augcov`: the fair version of `aug` (`CovariantAugment`, same ranges). One
  `k` per batch, undo `rho(g)` on the outputs, and multiply the physics
  residuals by `k`, so every loss term is evaluated in the sample's own frame.
  `k` is shared by the batch because the model stores one residual scalar per
  layer.
* `augmild`: as `aug` with `alpha ~ U(-0.5, 0.5)`, `k ~ logU(0.5, 2)`: keeps
  the physics-loss scale within 2x, isolating optimization difficulty from
  range coverage.
* `augphase` / `augscale`: the two axes of `augmild` separately (ablation of
  its in-distribution regularization effect).
* `canon`: same network wrapped in `Canonicalize`, with one frame per symmetry
  computed from the inputs only. Per graph it:
  * declares every transformer with `tap <= 1`;
  * subtracts the reference angle;
  * divides power-like inputs by `mean |Yff|`, relative to a constant fitted on
    the training set;
  * undoes the phase and scale frames on the outputs.

  Exact S1+S3+S4 equivariance by construction, no extra parameters. The
  orientation frame is the identity on case14 (taps 0.93-0.98). Runs trained
  before it existed are therefore valid in-distribution.

Protocol: random 80/10/10 scenario split per grid, normalizer fitted on the
train split (the framework's own `fit_on_train` convention), early stopping on
validation loss (patience 40, max 150 epochs), best-validation weights. With the
framework's usual patience of 15, the wide-augmentation run stops at epoch 16
with a validation loss 100x above the baseline (kept under
`results/case14_ieee/_protocol_p15/`). The longer patience gives every model the
same 150-epoch budget.

Reported per model and seed:
* in-distribution RMSE of the predicted quantities and power-balance residual;
* `EE_g` for `alpha in {0.1, 0.5, 1, pi}`, `k in {0.01, 0.1, 0.5, 2, 10, 100}`, a 50% line flip and a flip of all transformers;
* accuracy on each transformed test set `T_g`;
* zero-shot on `case30_ieee` and `case57_ieee`.

Zero-shot keeps the training grid's normalizer (`--zero-shot-norm source`,
recorded as `zero_shot_norm`). The framework default refits it on the target,
reading the target's `Qg` and slack `Pg`, which are PF outputs; it is still
available as `--zero-shot-norm refit`. Every `result.json` written before
2026-09-30 used refit.

## Caveats

* All local datasets use `theta_ref = 0`, so in-distribution `canon == m0` up
  to training noise (post hoc: 2e-7 relative). This PoC measures exactness and
  its cost, not transfer gains (Piano A H1 needs multi-grid pretraining).
* `canon` is exactly invariant to the normalizer's base, hence identical under
  both zero-shot protocols. M0 moves by up to 2x between them (seed 0).
* The `aug*` arms apply the augmentation per sample, `augcov` per batch.
* Training is on case14 only; the zero-shot grids are small IEEE cases.

## Cluster runs (abacus, from 2026-10-01)

Everything below runs through `cluster/*.sbatch` in the shared env (`source SYSTEMICO/env/activate.sh
<checkout>`); logs go to `cluster/logs/` (gitignored) and every job prints the sha1 of the code it ran, with a
content-addressed copy in `SymmetricalFM/provenance/` (`cluster/provenance.sh`).

**Case14 replicate** (`results/abacus/`: `PREREGISTRATION.md` before each batch, `VERDICTS.md`,
`prereg_output.txt` from `prereg.py`, tables in `case14_ieee/REPORT.md`). Training data: 2048 random scenarios
(seed 0) of HierarchicalFM's 10k-scenario datakit sets for case14/30/57 (`subset_data.py random`,
`cluster/build_train_data.sbatch`); the Mac PoC data and weights are not on the cluster, so this is a replicate on
independent draws. Arms added to the PoC's:

* `augcov` (E6b), `augcovphase` / `augcovscale` (its two axes), `augnophys` and the control `m0nophys` (E6a,
  physics loss weight 0);
* `canonmix` / `canonp95`: `Canonicalize(scale_a=a)` with the S3 frame s_a = P95(|input injections|)^a ·
  mean(|Yff|)^(1−a), a = 0.5 / 1 (Piano A H4; `canon` is a = 0);
* `m0warm`: M0 with canon's RNG path (one shuffled train pass before training);
* `m1canon` (F1, Piano A M1, `hodge.py`): `Canonicalize(BranchAngleHead(GNS))`, bus angles reconstructed by
  REF-anchored weighted least squares (weights |Y_ft|) from antisymmetric per-branch predictions on the backbone's
  final embeddings, PG/QG recomputed by the model's physics decoder; same frames as `canon`, so a difference
  between the two is one of representation.

`accuracy()` also returns the R7 split of the angle error: `VA_cm` (per-graph common offset of the predicted
angles relative to the clamped REF angle), `VA_diff` (the rest; VA² = VA_cm² + VA_diff²) and `dVA` (θ_f − θ_t,
each branch once). `report.ee_ratio` expresses the PG/QG error of an S3 probe in the original frame (× k).

**Released GENCO checkpoints** (F0, `GENCO_EXPERIMENTS.md` with the 2026-10-01 amendments):
`cluster/fetch_genco.sh` (login node: 36 IEEE checkpoints with their MLflow eval artifacts, full public datasets),
`cluster/build_genco_data.sbatch` (E0p on the full data, then `subset_data.py official`: every seed's full test
split plus the first 5000 train / 500 val ids, renumbered, with `splits/<size>_seed<s>.json`),
`cluster/genco_audit.sbatch` (per checkpoint, CPU: G1-G3 on the full official test split, refit gate, E1 raw and E2
post-hoc canon on 2000 test graphs, E5 on the three other grids with source / refit normalizer, the exact S3 probe
of the refit, canon under both), `cluster/split_audit.sbatch` (`split_audit.py`: exact duplicates and train/test
twins per released split). `genco_report.py` writes `results/genco/REPORT.md`.
