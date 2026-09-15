# AC power-flow symmetries: equivariance audit PoC

Proof of concept for [Piano A](../../AC_PF_SYMMETRIES_PLAN.md) (F0/F1): catalog the
exact symmetries of the AC-PF map, measure how much a GridFM PF surrogate breaks
them (equivariance error `EE_g`), and show that an exact fix costs nothing
in-distribution while augmentation only covers the range it was sampled from.

From the repository root, in an environment with `gridfm_graphkit` installed
(`.venv-screening` on the development machine):

```bash
.venv-screening/bin/python -m pytest -q experiments/ac_pf_symmetries/tests
.venv-screening/bin/python -m experiments.ac_pf_symmetries.run --variant m0 aug canon --seeds 0 1 2
.venv-screening/bin/python -m experiments.ac_pf_symmetries.report --case case14_ieee
```

## Symmetries implemented (`symmetries.py`)

Actions act on the normalized, masked hetero batch the model sees, and on the
output layout `[VM, VA, PG, QG]`.

| id | action on inputs | expected action on outputs |
|---|---|---|
| S1 phase | `VA_ref += alpha`, labels `VA += alpha` | `VA += alpha` |
| S3 scale | every power/admittance column `/= k` (P, Q, Q limits, GS, BS, Yff, Yft, rate), V untouched | `PG, QG /= k`; `VM, VA` invariant |
| S4 flip | swap the forward/reverse rows of a random subset of lines (tap = 1) | identity |

S2 (local gauge) needs phase shifters; none of the local datakit cases has one
(`shift == 0`, `Yft == Ytf` everywhere), so it is not testable here. S5 (node
permutation) holds for any GNN and is not measured.

`tests/test_symmetries.py` verifies on the repository fixture that (i) the
labelled states are PF solutions in the framework's own physics, (ii) the
power-balance residual of a *perturbed* state transforms exactly under each
action (invariance is checked on O(0.1) numbers, not on solver noise), (iii) the
output action matches the label action, (iv) a bare `GNS_heterogeneous` breaks
S1 and S3 but not S4, and (v) `Canonicalize` is equivariant to float precision.

## Models (`run.py`)

* `m0`: `GNS_heterogeneous` with the official `HGNS_PF_datakit_case14.yaml`
  hyper-parameters (12 layers, hidden 48, 8 heads, physics + masked-MSE loss).
* `aug`: same, trained with random S1 (`alpha ~ U(-pi, pi)`) and S3
  (`k ~ logU(0.1, 10)`) actions applied per sample ("wide": covers the
  audited range).
* `augmild`: as `aug` with `alpha ~ U(-0.5, 0.5)`, `k ~ logU(0.5, 2)`: keeps
  the physics-loss scale within 2x, isolating optimization difficulty from
  range coverage.
* `canon`: same network wrapped in `Canonicalize`: per graph, subtract the
  reference angle and divide power-like inputs by `mean |Yff|` (relative to a
  constant fitted on the training set), undo on the outputs. Exact S1+S3
  equivariance by construction, no extra parameters.

Protocol: random 80/10/10 scenario split per grid, normalizer fitted on the
train split (the framework's own `fit_on_train` convention, which also reads
`Pg`/`Qg` of the target grid), early stopping on validation loss (patience 40,
max 150 epochs), best-validation weights. With the framework's usual patience
of 15 the wide-augmentation run stops at epoch 16 with a validation loss 100x
above the baseline (kept under `results/case14_ieee/_protocol_p15/`); the
longer patience gives every model the same 150-epoch budget. Reported per model and seed:
in-distribution RMSE of the predicted quantities and power-balance residual;
`EE_g` for `alpha in {0.1, 0.5, 1, pi}`, `k in {0.01, 0.1, 0.5, 2, 10, 100}`
and a 50% line flip; accuracy on each transformed test set `T_g`; zero-shot on
`case30_ieee` and `case57_ieee` with their own fitted normalizers.

## Caveats

* All local datasets use `theta_ref = 0`, so in-distribution `canon == m0` up
  to training noise: this PoC measures exactness and its cost, not transfer
  gains (Piano A H1 needs multi-grid pretraining).
* Augmentation interacts with the physics loss: residuals scale like `1/k`, so
  the effective physics weight varies 100x across augmented samples.
* Training is on case14 only; the zero-shot grids are small IEEE cases.
